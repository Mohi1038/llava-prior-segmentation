"""Extra qualitative figures for the appendix (report v3). Reuses helpers and paths from make_figures.py.

Every image is chosen by a fixed rule (stated in the captions and logged in data/selection.json):
  f_class_gallery      one VOC val image per class: median-IoU object of that class among objects covering 5-60%
  f_ablation_examples  images at fixed quantiles of our per-image mIoU (>= 1 GT class), all ablation outputs
  f_layer_examples     same rule as f_class_gallery for 4 classes; maps-only vs SAM per layer range
  f_oracle_examples    images where oracle recognition helps (gain > 5 pts), at fixed quantiles of the gain
  f_mistral_person     person objects covering > 30% of the frame, at fixed quantiles of the Mistral - LLaVA-1.5 gain
Run: .venv/bin/python reports/neurips/make_extra_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

VOC, PRED, ROOT, DATA = mf.VOC, mf.PRED, mf.ROOT, mf.DATA
NAMES = mf.VOC_NAMES


def gt_of(image_id: str) -> np.ndarray:
    return mf.load_label(VOC / "SegmentationClass" / f"{image_id}.png")


def pred_of(d: Path, image_id: str) -> np.ndarray:
    return mf.load_label(d / f"{image_id}.png")


def score_tag(ax, val: float) -> None:
    ax.text(0.03, 0.04, f"{100 * val:.0f}", transform=ax.transAxes, fontsize=5.8, color="white",
            bbox={"boxstyle": "round,pad=0.15", "fc": "#000000aa", "ec": "none"})


def class_iou(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    keep = gt != 255
    p, g = (pred == c) & keep, (gt == c) & keep
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else float("nan")


def grid(rows: list[str], cols: list[tuple[str, object]], name: str, row_labels: list[str] | None = None,
         score: str = "miou", score_cls: list[int] | None = None, height: float = 1.08) -> None:
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(7.0, height * len(rows) + 0.25),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06}, squeeze=False)
    classes = set()
    for r, image_id in enumerate(rows):
        image = mf.load_rgb(VOC, image_id)
        gt = gt_of(image_id)
        classes |= set(mf.present(gt))
        for c, (title, src) in enumerate(cols):
            ax = axes[r, c]
            mf.blank(ax)
            if src is None:
                ax.imshow(image)
            elif src == "gt":
                ax.imshow(mf.overlay(image, gt))
            else:
                p = pred_of(src, image_id)
                classes |= set(mf.present(p))
                ax.imshow(mf.overlay(image, p))
                if score == "miou":
                    score_tag(ax, mf.per_image_score(p, gt, 21))
                elif score == "class" and score_cls is not None:
                    score_tag(ax, class_iou(p, gt, score_cls[r]))
            if r == 0:
                ax.set_title(title, fontsize=6.2, linespacing=1.0)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=6, color=mf.INK2, rotation=0, ha="right", va="center")
            ax.set_aspect("auto")
    mf.legend_classes(fig, sorted(classes), NAMES, y=0.075 if len(rows) > 3 else 0.06)
    mf.save(fig, name)


def objects() -> tuple[list[dict], list[str]]:
    st = mf.stats()
    return json.loads((DATA / "objects_voc.json").read_text()), st["voc_ids"]


def median_object(objs: list[dict], cls: int, lo: float = 0.05, hi: float = 0.6) -> dict | None:
    c = sorted([o for o in objs if o["cls"] == cls and lo <= o["frac"] <= hi], key=lambda o: (o["ours"], o["img"]))
    return c[len(c) // 2] if c else None


def fig_class_gallery(sel: dict) -> None:
    objs, ids = objects()
    picks = []
    for cls in range(1, 21):
        o = median_object(objs, cls)
        picks.append((cls, ids[o["img"]], o["ours"]))
    sel["class_gallery"] = {"rule": "per class: median-IoU object among objects covering 5-60% of the image",
                            "ids": [p[1] for p in picks]}
    base = PRED / "val_vocnames/base"
    fig, axes = plt.subplots(4, 10, figsize=(7.0, 3.35), gridspec_kw={"wspace": 0.04, "hspace": 0.3})
    for k, (cls, image_id, s) in enumerate(picks):
        image = mf.load_rgb(VOC, image_id)
        r, c = 2 * (k // 10), k % 10
        for rr, lab in ((r, gt_of(image_id)), (r + 1, pred_of(base, image_id))):
            ax = axes[rr, c]
            mf.blank(ax)
            ax.imshow(mf.overlay(image, lab))
            ax.set_aspect("auto")
        axes[r, c].set_title({"diningtable": "table", "pottedplant": "plant", "tvmonitor": "tv/monitor", "motorbike": "motorbike"}.get(NAMES[cls], NAMES[cls]), fontsize=6.0)
        score_tag(axes[r + 1, c], s)
        if c == 0:
            axes[r, c].set_ylabel("GT", fontsize=6, color=mf.INK2)
            axes[r + 1, c].set_ylabel("Ours", fontsize=6, color=mf.INK2)
    mf.save(fig, "f_class_gallery")


def fig_ablation_examples(sel: dict) -> None:
    st = mf.stats()
    ids = st["voc_ids"]
    pim = np.array(st["per_image_miou"]["ours"])
    order = list(np.argsort(pim))
    qs = [0.85, 0.6, 0.35]
    picks = [ids[order[int(q * (len(order) - 1))]] for q in qs]
    sel["ablation_examples"] = {"rule": "quantiles 85/60/35 of our per-image mIoU", "ids": picks}
    d = PRED / "val_vocnames"
    cols = [("Image\n", None), ("Ground\ntruth", "gt"), ("Full\n(65.9)", d / "base"), ("Points only\n(57.7)", d / "points_only"),
            ("Box only\n(54.1)", d / "box_only"), ("Product $D\\cdot M_A$\n(61.6)", d / "paper_product"),
            ("Token one early\n(48.6)", d / "shifted_token")]
    grid(picks, cols, "f_ablation_examples", [f"q{int(q * 100)}" for q in qs])


def fig_layer_examples(sel: dict) -> None:
    objs, ids = objects()
    picks = []
    for cname in ("dog", "car", "person", "bird"):
        o = median_object(objs, NAMES.index(cname), 0.1, 0.5)
        picks.append(ids[o["img"]])
    sel["layer_examples"] = {"rule": "per class (dog, car, person, bird): median-IoU object covering 10-50%",
                             "ids": picks}
    d = PRED / "val_layers"
    cols = [("Ground\ntruth", "gt"), ("Map\nL7-13", d / "map_layers_7_13"), ("SAM\nL7-13", d / "layers_7_13"),
            ("Map\nL15-22", d / "map_layers_15_22"), ("SAM\nL15-22", d / "layers_15_22"),
            ("Map\nL22-30", d / "map_layers_22_30"), ("SAM\nL22-30", d / "layers_22_30")]
    grid(picks, cols, "f_layer_examples")


def fig_oracle_examples(sel: dict) -> None:
    st = mf.stats()
    ids = st["voc_ids"]
    gain = np.array(st["per_image_miou"]["ours_oracle"]) - np.array(st["per_image_miou"]["ours"])
    cand = sorted([k for k in range(len(ids)) if gain[k] > 0.05], key=lambda k: (gain[k], ids[k]))
    qs = [0.9, 0.7, 0.5, 0.3]
    picks = [ids[cand[int(q * (len(cand) - 1))]] for q in qs]
    sel["oracle_examples"] = {"rule": f"images with oracle gain > 5 pts (n={len(cand)}), quantiles 90/70/50/30 of the gain",
                              "ids": picks}
    d = PRED / "val_vocnames"
    cols = [("Image\n", None), ("Ground\ntruth", "gt"), ("Maps\nonly", d / "base_coarse"), ("Ours\n(LLaVA answers)", d / "base"),
            ("Oracle\nmaps only", d / "qap_oracle_coarse"), ("Oracle\nrecognition", d / "qap_oracle")]
    grid(picks, cols, "f_oracle_examples", [f"q{int(q * 100)}" for q in qs], height=1.0)


def fig_mistral_person(sel: dict) -> None:
    objs, ids = objects()
    person = NAMES.index("person")
    cand = sorted([o for o in objs if o["cls"] == person and o["frac"] > 0.3],
                  key=lambda o: (o["v16m"] - o["ours"], o["img"]))
    qs = [0.9, 0.7, 0.5, 0.3]
    po = [cand[int(q * (len(cand) - 1))] for q in qs]
    picks = [ids[o["img"]] for o in po]
    sel["mistral_person"] = {"rule": f"person objects > 30% of frame (n={len(cand)}), quantiles 90/70/50/30 of "
                             "Mistral - LLaVA-1.5 person IoU", "ids": picks}
    cols = [("Image\n", None), ("Ground\ntruth", "gt"), ("LLaVA-1.5\nmaps only", PRED / "val_vocnames/base_coarse"),
            ("LLaVA-1.5\nfull", PRED / "val_vocnames/base"), ("1.6 Mistral\nmaps only", PRED / "v16m_val/base_coarse"),
            ("1.6 Mistral\nfull", PRED / "v16m_val/base")]
    grid(picks, cols, "f_mistral_person", [f"q{int(q * 100)}" for q in qs], score="class",
         score_cls=[person] * len(picks), height=1.0)


def main() -> None:
    sel_path = DATA / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.is_file() else {}
    for fn in (fig_class_gallery, fig_ablation_examples, fig_layer_examples, fig_oracle_examples, fig_mistral_person):
        fn(sel)
        print("done", fn.__name__)
    sel_path.write_text(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
