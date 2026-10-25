# Local datasets

Downloaded datasets belong in this directory and are intentionally excluded
from Git. Keep the original dataset archives and annotations intact.

## PASCAL VOC 2012

- Official dataset page: <https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/>
- Official train/validation archive:
  <https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>
- Archive filename: `VOCtrainval_11-May-2012.tar`
- Archive size: 1,999,639,040 bytes
- Verified SHA-256: `E14F763270CF193D0B5F74B169F44157A4B0C6EFA708F4DD0FF78EE691763BCB`
- Extracted root: `datasets/VOC2012/VOCdevkit/VOC2012`

The archive includes the images, class segmentation masks, and split lists.
Use `ImageSets/Segmentation/val.txt` for validation examples. VOC class masks
use the dataset's palette-indexed labels; value `255` denotes void/ignored pixels.
Retain the dataset's original class label IDs when preparing evaluation targets.

The local archive has been extracted and checked: it contains 17,125 JPEG images,
2,913 class-segmentation masks, and 1,449 validation IDs with matching image/mask
pairs. The three-image dog subset used to smoke-test batch inference is under
`VOCdevkit/batch-smoke-images`; its source IDs are recorded in `sample_ids.txt`.

## Curated all-class validation subset

Generate the deterministic class-balanced subset with:

```powershell
.\.venv\Scripts\llava-prior-prepare-voc.exe `
	.\datasets\VOC2012\VOCdevkit\VOC2012 `
	--output-dir .\datasets\VOC2012\curated `
	--samples-per-class 2
```

This selects the smallest-coverage and largest-coverage validation example for
each of the 20 classes. `curated/classes/<class>/image_ids.txt` lists examples
for each class; `curated/classes/<class>/images/` and `masks/` contain that
class's selected images and VOC annotations. `curated/images/` stores every
selected image once for shared-model batch inference,
`curated/class_map.json` maps image IDs to requested classes, and
`curated/manifest.json` records selection details. VOC ground-truth masks remain
in the original dataset tree. The current generated subset contains 40 unique
images, with two positive image/class examples per category.

The files are provided by the PASCAL VOC dataset maintainers. Consult the
official page and dataset terms before redistributing the archive, images,
annotations, or derived materials.
