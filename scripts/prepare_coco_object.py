"""Build COCO-Object (val2017) in the VOC folder layout used by ``llava-prior-voc``.

COCO-Object is the semantic-segmentation benchmark derived from COCO instance annotations
(used by GroupViT, TCL, SCLIP, ProxyCLIP and the FSeg-LLaVA paper): label 0 = background, labels
1..80 = the 80 object categories in COCO category-id order. Every instance mask is painted onto
the label map in annotation order (later instances overwrite earlier ones where they overlap);
crowd annotations are painted like other instances. Pixels covered by no instance are background.

Output (default datasets/coco_object):
  JPEGImages/<id>.jpg          symlinks to coco/val2017
  SegmentationClass/<id>.png   uint8 labels 0..80
  ImageSets/Segmentation/val.txt
  classes.txt                  the 80 COCO category names in label order

Usage: python scripts/prepare_coco_object.py [--coco datasets/coco] [--out datasets/coco_object]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

parser = argparse.ArgumentParser()
parser.add_argument("--coco", type=Path, default=Path("datasets/coco"))
parser.add_argument("--out", type=Path, default=Path("datasets/coco_object"))
args = parser.parse_args()

coco = COCO(str(args.coco / "annotations" / "instances_val2017.json"))
categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
assert len(categories) == 80
label_of = {c["id"]: index for index, c in enumerate(categories, start=1)}

for sub in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
    (args.out / sub).mkdir(parents=True, exist_ok=True)
(args.out / "classes.txt").write_text("\n".join(c["name"] for c in categories) + "\n")

ids = []
for image_id in sorted(coco.getImgIds()):
    info = coco.loadImgs(image_id)[0]
    stem = Path(info["file_name"]).stem
    label = np.zeros((info["height"], info["width"]), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=None)):
        label[coco.annToMask(ann) > 0] = label_of[ann["category_id"]]
    Image.fromarray(label).save(args.out / "SegmentationClass" / f"{stem}.png")
    link = args.out / "JPEGImages" / f"{stem}.jpg"
    if not link.exists():
        link.symlink_to((args.coco / "val2017" / info["file_name"]).resolve())
    ids.append(stem)

(args.out / "ImageSets" / "Segmentation" / "val.txt").write_text("\n".join(ids) + "\n")
print(f"{len(ids)} images, {len(categories)} classes -> {args.out}")
