"""All figures for the NeurIPS-format report, built from saved outputs (no GPU, no model loading).

Inputs: runs/pred/**/metrics.json, runs/official, runs/traces (LLaVA answers, response maps),
dataset images/GT, and reports/neurips/data/*.json from compute_stats.py.
Output: reports/neurips/figures/*.pdf (+ data/selection.json: which images each visual figure shows and why).

Run from the project root:  .venv/bin/python reports/neurips/make_figures.py [figure-name ...]
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import itertools

from llava_prior.activation import SegmentSettings, build_prompts, class_response
from llava_prior.report import voc_palette
from llava_prior.trace import ImageTrace

HERE = ROOT / "reports" / "neurips"
FIG = HERE / "figures"
DATA = HERE / "data"
PRED = ROOT / "runs" / "pred"
VOC = ROOT / "datasets" / "VOC2012" / "VOCdevkit" / "VOC2012"
COCO = ROOT / "datasets" / "coco_object"
VOC_NAMES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
             "tvmonitor"]
COCO_NAMES = ["background"] + (COCO / "classes.txt").read_text().split("\n")[:80]

# Reference palette (dataviz skill, light mode). Categorical slots in fixed order, one per system.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SYS = {"ours": "#2a78d6", "official": "#eb6834", "v16v": "#1baf7a", "v16m": "#eda100", "qwen": "#e87ba4"}
SYS_LABEL = {"ours": "Ours (LLaVA-1.5)", "official": "Official code (LLaVA-1.5)", "v16v": "LLaVA-1.6 Vicuna",
             "v16m": "LLaVA-1.6 Mistral", "qwen": "Qwen2.5-VL"}
# Ordinal blue ramp for pipeline stages (coarse -> full -> oracle recognition)
STAGE = {"coarse": "#86b6ef", "full": "#2a78d6", "oracle": "#104281"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
HEAT = "magma"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 2, "pdf.fonttype": 42, "savefig.dpi": 300,
    "legend.frameon": False, "axes.titleweight": "semibold", "axes.titlelocation": "left",
})
PAL = voc_palette()


def m(run: str) -> float | None:
    p = PRED / run / "metrics.json"
    if not p.is_file():
        p = ROOT / "runs" / run / "metrics.json"
    return 100 * json.loads(p.read_text())["mIoU"] if p.is_file() else None


def iou(run: str) -> dict[str, float]:
    p = PRED / run / "metrics.json"
    if not p.is_file():
        p = ROOT / "runs" / run / "metrics.json"
    return {k: (None if v is None else 100 * v) for k, v in json.loads(p.read_text())["IoU"].items()}


def stats() -> dict:
    return json.loads((DATA / "stats.json").read_text())


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)
    print("wrote", name)


def hgrid(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def vgrid(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- image helpers
def load_rgb(root: Path, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(root / "JPEGImages" / f"{image_id}.jpg").convert("RGB"))


def load_label(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def overlay(image: np.ndarray, label: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    color = PAL[label].astype(float)
    out = image.astype(float).copy()
    fg = (label > 0) & (label != 255)
    out[fg] = (1 - alpha) * out[fg] + alpha * color[fg]
    out[~fg] *= 0.55  # dim background so the masks read
    edge = np.zeros(label.shape, bool)
    edge[:-1] |= label[:-1] != label[1:]
    edge[:, :-1] |= label[:, :-1] != label[:, 1:]
    out[edge] = 255
    return out.astype(np.uint8)


def gt_view(label: np.ndarray) -> np.ndarray:
    return PAL[label]


def upsample(grid_map: np.ndarray, grid: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    t = Image.fromarray(grid_map.reshape(grid).astype(np.float32), mode="F")
    return np.asarray(t.resize((size[1], size[0]), Image.BILINEAR))


def heat_overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    h = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-6)
    color = plt.get_cmap(HEAT)(h)[..., :3] * 255
    return (0.35 * image + 0.65 * color).astype(np.uint8)


def blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def class_chips(ax, labels: list[int], names: list[str], y: float = -0.04) -> None:
    """Small colour chips + class names under an image panel (identity never colour-only)."""
    x = 0.0
    for lab in labels:
        c = PAL[lab] / 255
        ax.add_patch(Rectangle((x, y - 0.075), 0.06, 0.06, transform=ax.transAxes, color=c, clip_on=False))
        ax.text(x + 0.08, y - 0.045, names[lab], transform=ax.transAxes, fontsize=6, va="center", color=INK2)
        x += 0.1 + 0.055 * len(names[lab])


def present(label: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(label) if 0 < v < 255]


# ---------------------------------------------------------------- figure: pipeline walkthrough
def fig_pipeline(sel: dict) -> None:
    st = stats()
    ids = st["voc_ids"]
    pim = np.array(st["per_image_miou"]["ours"])
    # Deterministic choice: images with exactly two GT object classes, both answered "yes" by LLaVA,
    # per-image mIoU between the 70th and 80th percentile (a typical good case, not the best one).
    lo, hi = np.percentile(pim, [70, 80])
    chosen = None
    for k in np.argsort(-pim):
        if not lo <= pim[k] <= hi:
            continue
        gt = load_label(VOC / "SegmentationClass" / f"{ids[k]}.png")
        cls = present(gt)
        if len(cls) != 2:
            continue
        tr = ImageTrace.load(PRED.parent / "traces" / "val_vocnames", ids[k])
        if all((not tr.classes[VOC_NAMES[c]].forced) and "yes" in tr.classes[VOC_NAMES[c]].response.lower() for c in cls):
            chosen = ids[k]
            break
    image_id = chosen
    sel["pipeline"] = {"id": image_id, "rule": "2 GT classes, both recognised, per-image mIoU in the 70-80th percentile"}
    tr = ImageTrace.load(PRED.parent / "traces" / "val_vocnames", image_id)
    image = load_rgb(VOC, image_id)
    gt = load_label(VOC / "SegmentationClass" / f"{image_id}.png")
    names = VOC_NAMES[1:]
    s = SegmentSettings()
    prompts = build_prompts(tr, names, s)
    coarse = load_label(PRED / "val_vocnames/base_coarse" / f"{image_id}.png")
    final = load_label(PRED / "val_vocnames/base" / f"{image_id}.png")
    cls = present(gt)
    size = image.shape[:2]

    h, w = size
    fig = plt.figure(figsize=(7.0, 2 * 1.4 * h / w + 0.55))
    gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.05)
    # (a) input + recognition
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.imshow(image)
    ax.set_title("(a) Input image", fontsize=6.6)
    lines = []
    for n in names:
        e = tr.classes[n]
        yes = (not e.forced) and "yes" in e.response.lower()
        if yes or VOC_NAMES.index(n) in cls:
            lines.append((n, yes, e.response))
    n_no = sum(1 for n in names if not ((not tr.classes[n].forced) and "yes" in tr.classes[n].response.lower()))
    ax = fig.add_subplot(gs[1, 0])
    blank(ax)
    txt = "\n".join(f"{'✓' if y else '✗'} {n}: " + textwrap.fill(f"“{textwrap.shorten(r, 60, placeholder='…')}”", 24,
                                                                  subsequent_indent="   ") for n, y, r in lines[:3])
    ax.text(0.0, 1.02, "QAP, 20 questions: “Does the\nimage have the class of a\n<class>? …”\n" + txt +
            f"\n✗ {n_no} other classes: “No.”", transform=ax.transAxes, va="top", fontsize=5.0, color=INK2, linespacing=1.15)
    # (b) raw response maps D_c for the two classes, (c) reliable maps D_f
    for r, c in enumerate(cls[:2]):
        n = VOC_NAMES[c]
        raw = tr.classes[n].distance_aligned
        ax = fig.add_subplot(gs[r, 1])
        blank(ax)
        ax.imshow(heat_overlay(image, upsample(raw, tr.grid, size)))
        ax.set_title(f"({'b' if r == 0 else ' '}) $D_c$ {n}".replace("( ) ", ""), fontsize=6.6)
        df = class_response(tr.classes[n], tr.grid, s)
        ax = fig.add_subplot(gs[r, 2])
        blank(ax)
        ax.imshow(heat_overlay(image, upsample(df, tr.grid, size)))
        ax.set_title(f"({'c' if r == 0 else ' '}) $D_f$ {n}".replace("( ) ", ""), fontsize=6.6)
    # (d) coarse labels + SAM prompts
    ax = fig.add_subplot(gs[0, 3])
    blank(ax)
    ax.imshow(overlay(image, coarse))
    for p in prompts.prompts:
        col = PAL[p.slot] / 255
        if p.box is not None:
            x0, y0, x1, y1 = p.box
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.6))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=0.9))
        if len(p.points):
            ax.scatter(p.points[:, 0], p.points[:, 1], s=14, c=[col], edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_title("(d) SAM prompts", fontsize=6.6)
    ax = fig.add_subplot(gs[1, 3])
    blank(ax)
    ax.imshow(overlay(image, final))
    ax.set_title("(e) SAM output", fontsize=6.6)
    ax = fig.add_subplot(gs[0, 4])
    blank(ax)
    ax.imshow(overlay(image, gt))
    ax.set_title("(f) Ground truth", fontsize=6.6)
    ax = fig.add_subplot(gs[1, 4])
    blank(ax)
    err = np.zeros((*gt.shape, 3), np.uint8) + 255
    valid = gt != 255
    err[valid & (final == gt)] = (200, 220, 245)
    err[valid & (final != gt)] = (227, 73, 72)
    err[~valid] = (230, 230, 230)
    ax.imshow(err)
    ax.set_title("(g) Errors (red)", fontsize=6.6)
    save(fig, "f_pipeline")


# ---------------------------------------------------------------- figure: main results
def fig_main() -> None:
    st = stats()
    ci = st["ci"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.35})
    rows = [("ours", "ours_coarse", "ours", "ours_oracle"), ("official", None, "official", None),
            ("v16v", "v16v_coarse", "v16v", "v16v_oracle"), ("v16m", "v16m_coarse", "v16m", "v16m_oracle")]
    coarse_runs = {"ours_coarse": "val_vocnames/base_coarse", "v16v_coarse": "v16v_val/base_coarse",
                   "v16m_coarse": "v16m_val/base_coarse"}
    y = np.arange(len(rows))[::-1]
    for yi, (key, c, f, o) in zip(y, rows):
        vals = [m(coarse_runs[c]) if c else None, st["miou"][f], st["miou"][o] if o else None]
        pts = [(v, k) for v, k in zip(vals, ["coarse", "full", "oracle"]) if v is not None]
        a.plot([p[0] for p in pts], [yi] * len(pts), color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        lo, hi = ci[f]
        a.plot([lo, hi], [yi, yi], color=STAGE["full"], lw=1, zorder=2)
        for v, k in pts:
            a.scatter(v, yi, s=38, color=STAGE[k], edgecolor="white", linewidth=1.2, zorder=3)
        a.text(st["miou"][f], yi + 0.26, f"{st['miou'][f]:.2f}", ha="center", fontsize=6.5, color=INK)
        if o:
            a.text(st["miou"][o] + 0.8, yi, f"{st['miou'][o]:.1f}", va="center", fontsize=6, color=INK2)
        if c:
            a.text(vals[0], yi + 0.26, f"{vals[0]:.1f}", ha="center", fontsize=6, color=INK2)
    a.axvline(68.0, color=MUTED, lw=1, ls=(0, (3, 2)))
    a.text(68.0, len(rows) - 0.45, "paper 68.0", fontsize=6, color=INK2, ha="center")
    a.set_yticks(y, [SYS_LABEL[r[0]] for r in rows])
    a.set_xlim(10, 76)
    a.set_ylim(-0.6, len(rows) - 0.3)
    a.set_xlabel("VOC21 mIoU on PASCAL VOC 2012 val (1,449 images)")
    hgrid(a)
    a.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=STAGE[k], markersize=6, label=l)
                      for k, l in [("coarse", "LLaVA maps only (no SAM)"), ("full", "Full pipeline (95% CI)"),
                                   ("oracle", "Oracle recognition")]],
             loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=1)
    a.set_title("(a) VOC: reproduction and model transfer", pad=16)
    # COCO
    vals = [("coarse", m("coco_v15/base_coarse")), ("full", st["miou"]["coco"]), ("oracle", st["miou"]["coco_oracle"])]
    xs = np.arange(3)
    for x, (k, v) in zip(xs, vals):
        b.bar(x, v, width=0.6, color=STAGE[k], edgecolor="white", linewidth=1)
        b.text(x, v + 1, f"{v:.1f}", ha="center", fontsize=6.5)
    lo, hi = ci["coco"]
    b.plot([1, 1], [lo, hi], color=INK, lw=1)
    b.set_xticks(xs, ["Maps only", "Full", "Oracle\nrecognition"])
    b.set_ylabel("mIoU (81 classes)")
    b.set_ylim(0, 58)
    vgrid(b)
    b.set_title("(b) COCO-Object val (5,000 images)", pad=16)
    save(fig, "f_main")


# ---------------------------------------------------------------- figure: tuning-split model comparison
def fig_models_dev() -> None:
    rows = [("ours", "Ours (LLaVA-1.5)", m("dev_vocnames/base"), m("dev_vocnames/base_coarse")),
            ("official", "Official code (LLaVA-1.5)", m("official/dev200/bg"), None),
            ("v16v", "LLaVA-1.6 Vicuna", m("v16v_dev_layers/layers_7_13"), m("v16v_dev_layers/map_layers_7_13")),
            ("v16m", "LLaVA-1.6 Mistral (β 0.16)", m("v16m_dev_layers/beta0.16"), m("v16m_dev_layers/map_layers_7_13")),
            ("qwen", "Qwen2.5-VL-7B (best of grid)", m("qwen_dev_layers/b0.5_a0.9"), m("qwen_dev_layers/b0.45_a0.85_coarse"))]
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab, full, coarse) in zip(y, rows):
        ax.barh(yi, full, height=0.56, color=SYS[k], edgecolor="white")
        ax.text(full + 0.8, yi, f"{full:.1f}", va="center", fontsize=6.5)
        if coarse is not None:
            ax.plot([coarse, coarse], [yi - 0.28, yi + 0.28], color=INK, lw=1.2)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlim(0, 78)
    ax.set_xlabel("mIoU, 200-image tuning split (VOC train)")
    hgrid(ax)
    ax.legend(handles=[plt.Line2D([], [], color=INK, lw=1.2, marker="|", ls="", markersize=8, label="maps only (no SAM)")],
              loc="lower right", fontsize=6.5)
    save(fig, "f_models_dev")


# ---------------------------------------------------------------- figure: ablations
def fig_ablation() -> None:
    rows = [("No prototype step", "no_prototype", 67.3), ("No attention mask", "no_attention", 67.3),
            ("Paper-text product $D\\cdot M_A$", "paper_product", None), ("SAM points only", "points_only", 61.4),
            ("SAM box only", "box_only", 57.8), ("Class token read 1 position early", "shifted_token", None),
            ("No SAM (LLaVA maps only)", "base_coarse", None)]
    bv, bd = m("val_vocnames/base"), m("dev_vocnames/base")
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, run, paper) in zip(y, rows):
        dv = m(f"val_vocnames/{run}") - bv
        ax.barh(yi, dv, height=0.55, color=SYS["ours"], edgecolor="white")
        ax.text(10.5, yi, f"{dv:+.2f}", va="center", ha="right", fontsize=6.3, color=INK)
        dd = m(f"dev_vocnames/{run}") if run != "base_coarse" else m("dev_vocnames/base_coarse")
        if dd is not None:
            ax.scatter(dd - bd, yi + 0.0, marker="o", s=16, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=3)
        if paper is not None:
            ax.scatter(paper - 68.0, yi, marker="D", s=14, color=MUTED, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Δ mIoU vs full reproduction (val 65.85)")
    ax.set_xlim(-52, 11)
    ax.text(10.5, len(rows) - 0.35, "val Δ", ha="right", fontsize=6, color=INK2)
    hgrid(ax)
    ax.legend(bbox_to_anchor=(0, 1.02), ncol=3, columnspacing=0.8, handletextpad=0.3, handles=[Patch(color=SYS["ours"], label="val (1,449)"),
                       plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="tuning split (200)"),
                       plt.Line2D([], [], marker="D", ls="", color=MUTED, markersize=4, label="paper (Δ vs 68.0)")],
              loc="lower left", fontsize=6.2)
    save(fig, "f_ablation")


# ---------------------------------------------------------------- figure: layer study
def fig_layers() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.28})
    series = [("ours", "dev_layers", 33), ("v16v", "v16v_dev_layers", 33), ("v16m", "v16m_dev_layers", 33),
              ("qwen", "qwen_dev_layers", 29)]
    for k, d, n in series:
        vals = [m(f"{d}/layer_{i:02d}") for i in range(n)]
        xs = [i for i, v in enumerate(vals) if v is not None]
        a.plot(xs, [vals[i] for i in xs], color=SYS[k], lw=1.6, label={"ours": "LLaVA-1.5", "v16v": "LLaVA-1.6 Vicuna", "v16m": "LLaVA-1.6 Mistral", "qwen": "Qwen2.5-VL"}[k])
    a.axvspan(7, 13, color=GRID, alpha=0.7, lw=0)
    a.text(10, 1.5, "layers used\n(7–13)", ha="center", fontsize=6, color=INK2)
    a.set_xlabel("Language-model layer (0 = input embeddings)")
    a.set_ylabel("mIoU, single-layer map, no SAM")
    a.set_ylim(0, 42)
    vgrid(a)
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.2, ncol=4, handlelength=1.2, columnspacing=0.8)
    a.set_title("(a) Map quality per layer (tuning split)")
    ranges = ["5_9", "7_9", "7_13", "7_17", "10_13", "15_22", "22_30"]
    paper = {"5_9": 63.7, "7_9": 65.8, "7_13": 68.0, "7_17": 66.8, "10_13": 67.6, "15_22": 49.1, "22_30": 15.2}
    x = np.arange(len(ranges))
    w = 0.26
    for j, (k, d) in enumerate([("ours", "val_layers"), ("v16v", "v16v_dev_layers"), ("v16m", "v16m_dev_layers")]):
        vals = [m(f"{d}/layers_{r}") for r in ranges]
        b.bar(x + (j - 1) * w, vals, width=w, color=SYS[k], edgecolor="white", linewidth=0.6,
              label={"ours": "LLaVA-1.5 (val)", "v16v": "1.6 Vicuna (tuning)", "v16m": "1.6 Mistral (tuning)"}[k])
    b.scatter(x, [paper[r] for r in ranges], marker="D", s=14, color=INK, zorder=3, label="paper, LLaVA-1.5 (val)")
    b.set_xticks(x, [r.replace("_", "–") for r in ranges])
    b.set_xlabel("Layer range averaged for $D_c$")
    b.set_ylabel("mIoU with SAM")
    b.set_ylim(0, 80)
    b.tick_params(axis="x", labelsize=6.2)
    vgrid(b)
    b.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1, columnspacing=0.8)
    b.set_title("(b) Downstream quality with SAM", pad=24)
    a.set_title("(a) Map quality per layer (tuning split)", pad=24)
    save(fig, "f_layers")


def fig_layer_maps(sel: dict) -> None:
    """Per-layer response maps for one image/class, plus seed precision vs coverage."""
    st = stats()
    ids = st["voc_ids"]
    objs = json.loads((DATA / "objects_voc.json").read_text())
    # Deterministic: a single-object image (one GT class, 15-35% of the image), recognised, ours IoU >= 0.85.
    cands = [o for o in objs if 0.15 <= o["frac"] <= 0.35 and o["ours"] >= 0.85]
    one = {}
    for o in objs:
        one.setdefault(o["img"], []).append(o)
    cands = [o for o in cands if len(one[o["img"]]) == 1]
    animals = {VOC_NAMES.index(c) for c in ("bird", "cat", "cow", "dog", "horse", "sheep")}
    cands = [o for o in cands if o["cls"] in animals]
    cands.sort(key=lambda o: (-o["ours"], o["img"]))
    pick = cands[3]
    image_id, cname = ids[pick["img"]], VOC_NAMES[pick["cls"]]
    sel["layer_maps"] = {"id": image_id, "class": cname, "rule": "single-object animal image, 15-35% area, IoU>=0.85, 4th best"}
    tr = ImageTrace.load(PRED.parent / "traces" / "val_layers", image_id)
    image = load_rgb(VOC, image_id)
    gt = load_label(VOC / "SegmentationClass" / f"{image_id}.png")
    layers = [2, 5, 9, 13, 17, 21, 25, 31]
    ld = tr.classes[cname].layer_distance
    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(7.0, 1.05), gridspec_kw={"wspace": 0.04})
    blank(axes[0])
    axes[0].imshow(overlay(image, gt))
    axes[0].set_title(f"GT: {cname}", fontsize=6.5)
    for ax, l in zip(axes[1:], layers):
        blank(ax)
        ax.imshow(heat_overlay(image, upsample(ld[l], tr.grid, image.shape[:2])))
        ax.set_title(f"layer {l}", fontsize=6.5)
    save(fig, "f_layer_maps")


# ---------------------------------------------------------------- figure: qualitative grid
SYSTEM_DIRS = {"coarse": PRED / "val_vocnames/base_coarse", "ours": PRED / "val_vocnames/base",
               "official": ROOT / "runs/official/val/bg", "v16v": PRED / "v16v_val/base", "v16m": PRED / "v16m_val/base"}


def fig_qualitative(sel: dict) -> None:
    st = stats()
    ids = st["voc_ids"]
    pim = np.array(st["per_image_miou"]["ours"])
    order = np.argsort(pim)
    # Deterministic: images at fixed quantiles of our per-image mIoU, restricted to images with >= 2 GT classes
    # where possible so that more than one mask shows. Quantiles are stated in the caption.
    multi = [k for k in order if len(present(load_label(VOC / "SegmentationClass" / f"{ids[k]}.png"))) >= 2]
    qs = [0.95, 0.85, 0.7, 0.55, 0.4, 0.25]
    picks = [multi[int(q * (len(multi) - 1))] for q in qs]
    sel["qualitative"] = {"ids": [ids[k] for k in picks], "rule": "quantiles 95/85/70/55/40/25 of ours per-image mIoU "
                          "among images with >=2 GT classes", "per_image_miou": [pim[k] for k in picks]}
    cols = [("Image", None), ("Ground truth", "gt"), ("Maps only", "coarse"), ("Ours", "ours"),
            ("Official code", "official"), ("1.6 Vicuna", "v16v"), ("1.6 Mistral", "v16m")]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(7.0, 1.08 * len(picks) + 0.2),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    for r, k in enumerate(picks):
        image_id = ids[k]
        image = load_rgb(VOC, image_id)
        gt = load_label(VOC / "SegmentationClass" / f"{image_id}.png")
        for c, (title, key) in enumerate(cols):
            ax = axes[r, c]
            blank(ax)
            if key is None:
                ax.imshow(image)
                ax.set_ylabel(f"q{int(qs[r] * 100)}", fontsize=6, color=INK2, rotation=0, ha="right", va="center")
            elif key == "gt":
                ax.imshow(overlay(image, gt))
            else:
                ax.imshow(overlay(image, load_label(SYSTEM_DIRS[key] / f"{image_id}.png")))
                if key != "coarse":
                    s = per_image_score(load_label(SYSTEM_DIRS[key] / f"{image_id}.png"), gt, 21)
                    ax.text(0.03, 0.04, f"{100 * s:.0f}", transform=ax.transAxes, fontsize=5.8, color="white",
                            bbox={"boxstyle": "round,pad=0.15", "fc": "#000000aa", "ec": "none"})
            if r == 0:
                ax.set_title(title, fontsize=6.6)
            ax.set_aspect("auto")
    legend_classes(fig, sorted({c for k in picks for c in present(load_label(VOC / "SegmentationClass" / f"{ids[k]}.png"))}),
                   VOC_NAMES)
    save(fig, "f_qualitative")


def legend_classes(fig, classes: list[int], names: list[str], y: float = 0.085) -> None:
    handles = [Patch(color=PAL[c] / 255, label=names[c]) for c in classes]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=min(len(handles), 9), fontsize=6.2,
               handlelength=1, columnspacing=0.8)


def per_image_score(pred: np.ndarray, gt: np.ndarray, n: int) -> float:
    keep = gt != 255
    cm = np.bincount(n * gt[keep].astype(int) + pred[keep], minlength=n * n).reshape(n, n)
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    ok = union > 0
    return float(np.mean(tp[ok] / union[ok]))


# ---------------------------------------------------------------- figure: failure gallery
def fig_failures(sel: dict) -> None:
    st = stats()
    ids = st["voc_ids"]
    objs = json.loads((DATA / "objects_voc.json").read_text())
    traces = PRED.parent / "traces" / "val_vocnames"
    cases = []
    answers: dict[str, dict] = {}

    def kept(k: int, c: int) -> bool:
        """LLaVA's greedy answer was yes (a 'forced' answer means the greedy answer was 'No.')."""
        if ids[k] not in answers:
            answers[ids[k]] = json.loads((traces / f"{ids[k]}.json").read_text())["classes"]
        e = answers[ids[k]][VOC_NAMES[c]]
        return (not e["forced"]) and "yes" in e["response"].lower()

    per_img: dict[int, list] = {}
    for o in objs:
        per_img.setdefault(o["img"], []).append(o)
    # 1) Close-up person: person is the only GT class, covers > 45% of the image, recognised, IoU < 0.3.
    #    Median-severity case of that set (representative, not the most extreme).
    person = VOC_NAMES.index("person")
    c1 = sorted([o for o in objs if o["cls"] == person and o["frac"] > 0.45 and o["ours"] < 0.3
                 and len(per_img[o["img"]]) == 1 and kept(o["img"], person)], key=lambda o: (o["ours"], o["img"]))
    cases.append(("Close-up: prior covers only part of the person", c1[len(c1) // 2]["img"], person))
    # 2) Recognition miss: LLaVA answered no, oracle recognition recovers it (oracle IoU >= 0.7); largest such object.
    c2 = sorted([o for o in objs if o["ours"] < 0.05 and o["ours_oracle"] >= 0.7 and not kept(o["img"], o["cls"])],
                key=lambda o: (-o["frac"], o["img"]))
    cases.append(("Missed class: LLaVA answered “No.”", c2[0]["img"], c2[0]["cls"]))
    # 3) Hallucinated class: largest predicted area of a class absent from the GT.
    best = None
    for k, image_id in enumerate(ids):
        if k % 3:
            continue  # subsample for speed; deterministic
        pred = load_label(PRED / "val_vocnames/base" / f"{image_id}.png")
        gt = load_label(VOC / "SegmentationClass" / f"{image_id}.png")
        g = set(present(gt))
        for c in present(pred):
            if c not in g:
                frac = float((pred == c).mean())
                if best is None or frac > best[0]:
                    best = (frac, k, c)
    cases.append(("False “yes”: absent class segmented", best[1], best[2]))
    # 4) Small/thin objects: GT class < 2% of the image, recognised, IoU < 0.2 (boundary/size limit).
    used = {k for _, k, _ in cases}
    c4 = sorted([o for o in objs if o["frac"] < 0.02 and o["ours"] < 0.2 and o["cls"] != person
                 and o["img"] not in used and kept(o["img"], o["cls"])], key=lambda o: (o["frac"], o["img"]))
    cases.append(("Small object: 24×24 prior too coarse", c4[len(c4) // 2]["img"], c4[len(c4) // 2]["cls"]))
    sel["failures"] = [{"case": t, "id": ids[k], "class": VOC_NAMES[c]} for t, k, c in cases]

    fig, axes = plt.subplots(len(cases), 4, figsize=(7.0, 1.42 * len(cases)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.35], "wspace": 0.04, "hspace": 0.12})
    for r, (title, k, c) in enumerate(cases):
        image_id = ids[k]
        image = load_rgb(VOC, image_id)
        gt = load_label(VOC / "SegmentationClass" / f"{image_id}.png")
        pred = load_label(PRED / "val_vocnames/base" / f"{image_id}.png")
        tr = ImageTrace.load(traces, image_id)
        name = VOC_NAMES[c]
        e = tr.classes[name]
        for ax in axes[r]:
            blank(ax)
            ax.set_aspect("auto")
        axes[r, 0].imshow(overlay(image, gt))
        axes[r, 1].imshow(overlay(image, pred))
        if e.distance_aligned is not None:
            axes[r, 2].imshow(heat_overlay(image, upsample(e.distance_aligned, tr.grid, image.shape[:2])))
        else:
            axes[r, 2].imshow(image)
            axes[r, 2].text(0.5, 0.5, "no map\n(answer: no)", ha="center", va="center", transform=axes[r, 2].transAxes,
                            color="white", fontsize=7, bbox={"fc": "#000000aa", "ec": "none"})
        if r == 0:
            for ax, t in zip(axes[r], ["Ground truth", "Ours", "Response map $D_c$", "LLaVA's answer (verbatim)"]):
                ax.set_title(t, fontsize=6.8)
        axes[r, 0].set_ylabel(f"({chr(97 + r)})", rotation=0, ha="right", va="center", fontsize=7)
        ans = e.response.strip().replace("\n", " ")
        # A 'forced' entry means LLaVA's greedy answer was "No."; the stored text is a forced "yes"
        # continuation generated only to obtain a map for analysis.
        shown = "No." if e.forced else ans
        verdict = "kept" if (not e.forced and "yes" in ans.lower()) else "rejected"
        body = textwrap.fill(f"Q: “Does the image have the class of a {name}? Answer yes or no …”", 44) + "\n" + \
            textwrap.fill(f"A: “{textwrap.shorten(shown, 150, placeholder='…')}”", 44)
        if e.forced:
            body += "\n" + textwrap.fill("(map shown from a forced “yes” continuation, analysis only)", 50)
        ax = axes[r, 3]
        ax.text(0.03, 0.97, title, transform=ax.transAxes, va="top", fontsize=6.6, weight="semibold")
        ax.text(0.03, 0.80, body, transform=ax.transAxes, va="top", fontsize=6, color=INK2)
        ax.text(0.03, 0.06, f"class {verdict} · P(yes) = {e.p_yes:.2f} · GT: " +
                ", ".join(VOC_NAMES[x] for x in present(gt)), transform=ax.transAxes, fontsize=5.6, color=MUTED)
    save(fig, "f_failures")


# ---------------------------------------------------------------- figure: per-class heatmap
def fig_per_class() -> None:
    runs = [("Maps only", "val_vocnames/base_coarse"), ("Ours", "val_vocnames/base"), ("Official", "official/val/bg"),
            ("1.6 Vicuna", "v16v_val/base"), ("1.6 Mistral", "v16m_val/base"), ("Ours + oracle\nrecognition", "val_vocnames/qap_oracle")]
    mats = np.array([[iou(r)[n] for n in VOC_NAMES] for _, r in runs])
    order = np.argsort(-mats[1])
    mats = mats[:, order]
    names = [VOC_NAMES[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.0, 1.95))
    im = ax.imshow(mats, cmap=BLUES, vmin=0, vmax=100, aspect="auto")
    for i in range(mats.shape[0]):
        for j in range(mats.shape[1]):
            v = mats[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6, color="white" if v > 55 else INK)
    ax.set_xticks(range(len(names)), [n.replace("diningtable", "dining table").replace("pottedplant", "potted plant")
                                      .replace("tvmonitor", "tv/monitor") for n in names], rotation=40, ha="right")
    ax.set_yticks(range(len(runs)), [r[0] for r in runs])
    ax.axhline(0.5, color="white", lw=2)
    ax.axhline(4.5, color="white", lw=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("IoU (%)", fontsize=6.5)
    cb.outline.set_visible(False)
    save(fig, "f_per_class")


# ---------------------------------------------------------------- figure: recognition
def fig_recognition() -> None:
    st = stats()
    q = st["qap"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.3})
    pc = q["ours"]["per_class"]
    names = sorted(pc, key=lambda n: pc[n]["precision"])
    x = np.arange(len(names))
    for xi, n in zip(x, names):
        a.plot([xi, xi], [pc[n]["precision"], pc[n]["recall"]], color=GRID, lw=2, zorder=1)
    a.scatter(x, [pc[n]["precision"] for n in names], s=22, color=SYS["official"], label="precision", zorder=3,
              edgecolor="white", linewidth=0.8)
    a.scatter(x, [pc[n]["recall"] for n in names], s=22, color=SYS["ours"], label="recall", zorder=3,
              edgecolor="white", linewidth=0.8)
    for xi, n in zip(x, names):
        a.text(xi, pc[n]["precision"] - 0.06, f"{int(pc[n]['fp'])}", ha="center", fontsize=5.2, color=MUTED)
    a.set_xticks(x, [n.replace("diningtable", "dining table").replace("pottedplant", "potted plant")
                     .replace("tvmonitor", "tv/monitor") for n in names], rotation=40, ha="right")
    a.set_ylim(0.3, 1.04)
    a.set_ylabel("LLaVA-1.5 yes/no answers")
    vgrid(a)
    a.legend(loc="lower right", ncol=2)
    a.set_title("(a) LLaVA-1.5 recognition per class, VOC val")
    models = [("ours", "1.5\nVOC"), ("v16v", "1.6 Vic.\nVOC"), ("v16m", "1.6 Mis.\nVOC"), ("coco", "1.5\nCOCO")]
    w = 0.38
    xm = np.arange(len(models))
    P = [q[k]["greedy"]["precision"] for k, _ in models]
    R = [q[k]["greedy"]["recall"] for k, _ in models]
    b.bar(xm - w / 2, P, width=w, color=SYS["official"], edgecolor="white", label="precision")
    b.bar(xm + w / 2, R, width=w, color=SYS["ours"], edgecolor="white", label="recall")
    for xi, p, r in zip(xm, P, R):
        b.text(xi - w / 2, p + 0.015, f"{p:.2f}", ha="center", fontsize=5.6)
        b.text(xi + w / 2, r + 0.015, f"{r:.2f}", ha="center", fontsize=5.6)
    b.set_xticks(xm, [mm[1] for mm in models], fontsize=6)
    b.set_ylim(0, 1.12)
    vgrid(b)
    b.set_title("(b) Overall")
    save(fig, "f_recognition")


# ---------------------------------------------------------------- figure: error decomposition + size
def fig_errors() -> None:
    st = stats()
    e = st["errors"]
    rows = [("Maps only (VOC)", "ours_coarse"), ("Ours (VOC)", "ours"), ("Official (VOC)", "official"),
            ("1.6 Vicuna (VOC)", "v16v"), ("1.6 Mistral (VOC)", "v16m"), ("Ours + oracle (VOC)", "ours_oracle"),
            ("Ours (COCO)", "coco"), ("Ours + oracle (COCO)", "coco_oracle")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55})
    y = np.arange(len(rows))[::-1]
    parts = [("fg_correct", "correct class", SYS["ours"]), ("fg_to_background", "missed (→ background)", "#9ec5f4"),
             ("fg_wrong_class", "wrong object class", SYS["official"])]
    for yi, (lab, k) in zip(y, rows):
        left = 0
        for key, _, col in parts:
            v = 100 * e[k][key]
            a.barh(yi, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=1)
            if v > 7:
                a.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=6,
                       color="white" if col != "#9ec5f4" else INK)
            left += v
        a.text(101.5, yi, f"{100 * e[k]['bg_marked_fg_per_fg']:.0f}", va="center", fontsize=6, color=INK2)
    a.text(101.5, len(rows) - 0.35, "false fg*", fontsize=5.8, color=INK2)
    a.set_yticks(y, [r[0] for r in rows])
    a.set_xlim(0, 100)
    a.set_xlabel("Share of ground-truth object pixels (%)")
    a.legend(handles=[Patch(color=c, label=l) for _, l, c in parts], loc="lower left", bbox_to_anchor=(0, 1.0),
             ncol=3, fontsize=6.2, handlelength=1, columnspacing=0.8)
    a.set_title("(a) Where object pixels go", pad=14)
    # size analysis
    for ds, f, keys in [("VOC", "objects_voc.json", [("ours_coarse", "coarse"), ("ours", "full"), ("ours_oracle", "oracle")]),
                        ("COCO", "objects_coco.json", [("coco_coarse", "coarse"), ("coco", "full"), ("coco_oracle", "oracle")])]:
        objs = json.loads((DATA / f).read_text())
        bins = [0, 0.01, 0.03, 0.1, 0.3, 1.01]
        frac = np.array([o["frac"] for o in objs])
        for k, stage in keys:
            v = np.array([o[k] for o in objs])
            means = [100 * v[(frac >= lo) & (frac < hi)].mean() for lo, hi in itertools.pairwise(bins)]
            b.plot(range(len(means)), means, color=STAGE[stage], lw=1.6 if ds == "VOC" else 1.2,
                   ls="-" if ds == "VOC" else (0, (3, 1.5)), marker="o" if ds == "VOC" else "s", markersize=3.5,
                   markeredgecolor="white", markeredgewidth=0.6)
    b.set_xticks(range(5), ["<1%", "1–3%", "3–10%", "10–30%", ">30%"])
    b.set_xlabel("Object size (share of image area)")
    b.set_ylabel("mean per-object IoU (%)")
    vgrid(b)
    b.set_ylim(0, 92)
    hs = [plt.Line2D([], [], color=STAGE[s], lw=1.6, label=l) for s, l in
          [("coarse", "maps only"), ("full", "full pipeline"), ("oracle", "oracle recognition")]]
    hs += [plt.Line2D([], [], color=INK2, lw=1.4, marker="o", markersize=3, label="VOC"),
           plt.Line2D([], [], color=INK2, lw=1.2, ls=(0, (3, 1.5)), marker="s", markersize=3, label="COCO")]
    b.legend(handles=hs, fontsize=5.8, loc="upper left", ncol=2, handlelength=1.6)
    b.set_title("(b) Accuracy by object size", pad=14)
    save(fig, "f_errors")


# ---------------------------------------------------------------- figure: COCO per class + examples
def fig_coco(sel: dict) -> None:
    base, orc = iou("coco_v15/base"), iou("coco_v15/qap_oracle")
    st = stats()
    q = st["qap"]["coco"]["per_class"]
    names = COCO_NAMES[1:]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.3})
    xs = np.array([100 * q[n]["precision"] for n in names])
    ys = np.array([orc[n] - base[n] for n in names])
    a.scatter(xs, ys, s=14, color=SYS["ours"], edgecolor="white", linewidth=0.5)
    for i in np.argsort(-ys)[:7]:
        a.annotate(names[i], (xs[i], ys[i]), xytext=(3, 1), textcoords="offset points", fontsize=5.6, color=INK2)
    r_ = np.corrcoef(xs, ys)[0, 1]
    a.text(0.97, 0.95, f"Pearson r = {r_:.2f}", transform=a.transAxes, ha="right", va="top", fontsize=6.2, color=INK2)
    a.set_xlabel("Recognition precision of the class (%)")
    a.set_ylabel("IoU gain from oracle recognition")
    vgrid(a)
    a.set_title("(a) Oracle gain vs. false “yes” answers")
    order = np.argsort([base[n] for n in names])
    show = list(order[:12]) + list(order[-12:])
    yy = np.arange(len(show))
    for yi, i in zip(yy, show):
        n = names[i]
        b.plot([base[n], orc[n]], [yi, yi], color=GRID, lw=2, zorder=1)
        b.scatter(base[n], yi, s=16, color=STAGE["full"], zorder=3, edgecolor="white", linewidth=0.5)
        b.scatter(orc[n], yi, s=16, color=STAGE["oracle"], zorder=3, edgecolor="white", linewidth=0.5)
    b.axhline(11.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    b.set_yticks(yy, [names[i] for i in show], fontsize=5.4)
    b.set_xlabel("IoU (%)")
    hgrid(b)
    b.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=STAGE[s], label=l, markersize=4)
                      for s, l in [("full", "full pipeline"), ("oracle", "oracle recognition")]],
             loc="lower right", fontsize=6)
    b.set_title("(b) 12 weakest and 12 strongest COCO classes")
    save(fig, "f_coco_classes")

    # qualitative COCO: quantiles of per-image mIoU among images with >= 3 GT classes
    ids = st["coco_ids"]
    pim = np.array(st["per_image_miou"]["coco"])
    order = np.argsort(pim)
    multi = [k for k in order if len(present(load_label(COCO / "SegmentationClass" / f"{ids[k]}.png"))) >= 3]
    qs = [0.9, 0.7, 0.5, 0.3]
    picks = [multi[int(q_ * (len(multi) - 1))] for q_ in qs]
    sel["coco_examples"] = {"ids": [ids[k] for k in picks], "rule": "quantiles 90/70/50/30 of per-image mIoU, >=3 GT classes"}
    cols = [("Image", None), ("Ground truth", "gt"), ("Maps only", "coco_v15/base_coarse"), ("Full pipeline", "coco_v15/base"),
            ("Oracle recognition", "coco_v15/qap_oracle")]
    fig, axes = plt.subplots(len(cols), len(picks), figsize=(7.0, 5.0), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    allc = set()
    for c, k in enumerate(picks):
        image_id = ids[k]
        image = load_rgb(COCO, image_id)
        gt = load_label(COCO / "SegmentationClass" / f"{image_id}.png")
        allc |= set(present(gt))
        for r, (title, key) in enumerate(cols):
            ax = axes[r, c]
            blank(ax)
            ax.set_aspect("auto")
            if key is None:
                ax.imshow(image, aspect="auto")
            elif key == "gt":
                ax.imshow(overlay(image, gt), aspect="auto")
            else:
                p = load_label(PRED / key / f"{image_id}.png")
                allc |= set(present(p))
                ax.imshow(overlay(image, p), aspect="auto")
                if "coarse" not in key:
                    ax.text(0.03, 0.05, f"{100 * per_image_score(p, gt, 81):.0f}", transform=ax.transAxes, fontsize=6,
                            color="white", bbox={"boxstyle": "round,pad=0.15", "fc": "#000000aa", "ec": "none"})
            if c == 0:
                ax.set_ylabel(title, fontsize=6.6)
    handles = [Patch(color=PAL[x] / 255, label=COCO_NAMES[x]) for x in sorted(allc)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.075), ncol=8, fontsize=5.8, handlelength=1,
               columnspacing=0.7)
    save(fig, "f_coco_examples")


# ---------------------------------------------------------------- figure: improvement attempts
def fig_attempts() -> None:
    st = stats()
    bd = m("dev_vocnames/base")
    rows = [("Evidence filter (unsure “yes”)", "dev_vocnames/filter_p0.7_a0.02"),
            ("Per-class background threshold", "dev_vocnames/bg_per_class"),
            ("Relative background (0.7)", "dev_vocnames/bg_rel0.7"),
            ("One SAM prompt per region", "dev_vocnames/comp_f0.01"),
            ("Query alias “person, people”", "dev_person_people/base"),
            ("Calibrated P(yes) ≥ 0.65", "dev_vocnames/qap_p0.65"),
            ("Coverage box from layers 12–17", "dev_layers/cov_12_17_a0.6"),
            ("Median-peak normalisation", "dev_layers/norm_a0.5_b0.6"),
            ("Centred hidden states (best)", "dev_c/c_norm_a0.75_l16_22"),
            ("Layers 7–17 instead of 7–13", "dev_layers/layers_7_17"),
            ("Combination (3 changes)", "dev_vocnames/combo_small")]
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, run) in zip(y, rows):
        d = m(run) - bd
        col = SYS["ours"] if d > 0 else "#b7d3f6"
        ax.barh(yi, max(d, -6), height=0.58, color=col, edgecolor="white")
        if d <= -6:
            ax.text(-5.9, yi, f"◂ {d:+.1f} (axis cut)", va="center", ha="left", fontsize=5.6, color=INK)
        else:
            ax.text((d - 0.15) if d < 0 else d + 0.15, yi, f"{d:+.2f}", va="center", ha="right" if d < 0 else "left",
                    fontsize=5.8, color=INK)
    p = st["paired"]["ours->combo"]
    yc = y[-1]
    ax.errorbar(p["delta"], yc - 0.42, xerr=[[p["delta"] - p["ci95"][0]], [p["ci95"][1] - p["delta"]]], fmt="D",
                color=SYS["official"], markersize=3.5, lw=1, capsize=2)
    ax.text(p["ci95"][1] + 0.15, yc - 0.42, f"val {p['delta']:+.2f}", va="center", fontsize=5.8, color=SYS["official"])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.2)
    ax.set_xlim(-6.5, 3.2)
    ax.set_xlabel("Δ mIoU vs reproduction, tuning split (68.49)")
    hgrid(ax)
    ax.legend(handles=[Patch(color=SYS["ours"], label="tuning gain"), Patch(color="#b7d3f6", label="tuning loss"),
                       plt.Line2D([], [], marker="D", color=SYS["official"], ls="", markersize=3.5, label="val, 95% CI")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=5.8, columnspacing=0.8, handletextpad=0.3)
    save(fig, "f_attempts")


# ---------------------------------------------------------------- figure: implementation agreement
def fig_agreement() -> None:
    st = stats()
    a_ = np.array(st["per_image_miou"]["ours"])
    o_ = np.array(st["per_image_miou"]["official"])
    fig, ax = plt.subplots(figsize=(2.3, 2.2))
    ax.hexbin(100 * o_, 100 * a_, gridsize=28, cmap=BLUES, mincnt=1, bins="log", linewidths=0.2, edgecolors="white")
    ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    r = np.corrcoef(a_, o_)[0, 1]
    within = np.mean(np.abs(a_ - o_) < 0.05)
    ax.text(3, 93, f"Pearson r = {r:.2f}\n{100 * within:.0f}% of images within 5 pts", fontsize=6, va="top")
    ax.set_xlabel("Official code, per-image mIoU")
    ax.set_ylabel("Ours, per-image mIoU")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    save(fig, "f_agreement")


def write_numbers() -> None:
    """Every statistic quoted in the report text, as LaTeX macros (data/numbers.tex)."""
    st = stats()
    out = []

    def mac(name: str, value: str) -> None:
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for k, n in [("ours", "Ours"), ("official", "Official"), ("v16v", "Vic"), ("v16m", "Mis"), ("coco", "Coco"),
                 ("ours_oracle", "OursOracle"), ("coco_oracle", "CocoOracle"), ("ours_coarse", "OursCoarse")]:
        mac(f"miou{n}", f"{st['miou'][k]:.2f}")
        lo, hi = st["ci"][k]
        mac(f"ci{n}", f"[{lo:.1f}, {hi:.1f}]")
    for key, n in [("ours->official", "Official"), ("ours->v16v", "Vic"), ("ours->v16m", "Mis"),
                   ("ours->ours_oracle", "Oracle"), ("ours->combo", "Combo"), ("coco->coco_oracle", "CocoOracle"),
                   ("v16v->v16m", "VicMis")]:
        p = st["paired"][key]
        mac(f"d{n}", f"{p['delta']:+.2f}")
        mac(f"dci{n}", f"[{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]")
    for k, n in [("ours", "Ours"), ("ours_coarse", "OursCoarse"), ("ours_oracle", "OursOracle"), ("coco", "Coco"),
                 ("coco_oracle", "CocoOracle"), ("v16m", "Mis"), ("v16v", "Vic"), ("official", "Official")]:
        e = st["errors"][k]
        mac(f"eCorrect{n}", f"{100 * e['fg_correct']:.1f}")
        mac(f"eBg{n}", f"{100 * e['fg_to_background']:.1f}")
        mac(f"eWrong{n}", f"{100 * e['fg_wrong_class']:.1f}")
        mac(f"eFalse{n}", f"{100 * e['bg_marked_fg_per_fg']:.1f}")
    for k, n in [("ours", "Ours"), ("v16v", "Vic"), ("v16m", "Mis"), ("coco", "Coco")]:
        g = st["qap"][k]["greedy"]
        mac(f"qP{n}", f"{g['precision']:.3f}")
        mac(f"qR{n}", f"{g['recall']:.3f}")
        mac(f"qFP{n}", f"{int(g['fp']):,}")
        mac(f"qFN{n}", f"{int(g['fn']):,}")
    a_ = np.array(st["per_image_miou"]["ours"])
    o_ = np.array(st["per_image_miou"]["official"])
    mac("agreeR", f"{np.corrcoef(a_, o_)[0, 1]:.2f}")
    mac("agreeWithin", f"{100 * np.mean(np.abs(a_ - o_) < 0.05):.0f}")
    for ds, f, keys in [("Voc", "objects_voc.json", ["ours", "ours_oracle", "ours_coarse"]),
                        ("Coco", "objects_coco.json", ["coco", "coco_oracle", "coco_coarse"])]:
        objs = json.loads((DATA / f).read_text())
        frac = np.array([o["frac"] for o in objs])
        for k in keys:
            v = np.array([o[k] for o in objs])
            tag = {"ours": "", "coco": "", "ours_oracle": "Oracle", "coco_oracle": "Oracle", "ours_coarse": "Coarse",
                   "coco_coarse": "Coarse"}[k]
            mac(f"size{ds}{tag}Small", f"{100 * v[frac < 0.01].mean():.1f}")
            mac(f"size{ds}{tag}Large", f"{100 * v[frac >= 0.3].mean():.1f}")
        mac(f"size{ds}SmallShare", f"{100 * np.mean(frac < 0.01):.0f}")
    q = st["qap"]["coco"]["per_class"]
    base, orc = iou("coco_v15/base"), iou("coco_v15/qap_oracle")
    nm = COCO_NAMES[1:]
    gain = np.array([orc[n] - base[n] for n in nm])
    mac("cocoGainPrecR", f"{np.corrcoef(gain, [q[n]['precision'] for n in nm])[0, 1]:.2f}")
    mac("cocoGainRecR", f"{np.corrcoef(gain, [q[n]['recall'] for n in nm])[0, 1]:.2f}")
    (DATA / "numbers.tex").write_text("\n".join(out) + "\n")
    print("wrote numbers.tex", len(out))


def main() -> None:
    want = set(sys.argv[1:])
    sel_path = DATA / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.is_file() else {}
    figs = {"main": lambda: fig_main(), "models_dev": lambda: fig_models_dev(), "ablation": lambda: fig_ablation(),
            "layers": lambda: fig_layers(), "layer_maps": lambda: fig_layer_maps(sel),
            "pipeline": lambda: fig_pipeline(sel), "qualitative": lambda: fig_qualitative(sel),
            "failures": lambda: fig_failures(sel), "per_class": lambda: fig_per_class(),
            "recognition": lambda: fig_recognition(), "errors": lambda: fig_errors(), "coco": lambda: fig_coco(sel),
            "attempts": lambda: fig_attempts(), "agreement": lambda: fig_agreement(), "numbers": write_numbers}
    for name, fn in figs.items():
        if not want or name in want:
            fn()
    sel_path.write_text(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
