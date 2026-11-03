"""Compare two trace directories image by image (self-check for tracer speed-ups).

Usage: python scripts/compare_traces.py <reference traces> <new traces>
Reports answer agreement, P(yes) differences and the largest map difference on shared images.
"""

import json
import sys
from pathlib import Path

import numpy as np

from llava_prior.trace import ImageTrace

reference, candidate = Path(sys.argv[1]), Path(sys.argv[2])
ids = sorted(p.stem for p in candidate.glob("*.npz"))
answers = same_answer = 0
p_diff, map_diff = [], []
for image_id in ids:
    a, b = ImageTrace.load(reference, image_id), ImageTrace.load(candidate, image_id)
    seconds = (a.settings.get("seconds"), b.settings.get("seconds"))
    for name, ca in a.classes.items():
        cb = b.classes[name]
        yes_a = not ca.forced and "yes" in ca.response.lower()
        yes_b = not cb.forced and "yes" in cb.response.lower()
        answers += 1
        same_answer += yes_a == yes_b
        p_diff.append(abs(ca.p_yes - cb.p_yes))
        if ca.distance_aligned is not None and cb.distance_aligned is not None:
            map_diff.append(float(np.abs(ca.distance_aligned - cb.distance_aligned).max()))
times = [ImageTrace.load(d, i).settings.get("seconds", float("nan")) for d in (reference, candidate) for i in ids]
half = len(ids)
print(json.dumps({
    "images": len(ids),
    "answers": answers,
    "same_yes_no": same_answer,
    "max_p_yes_diff": max(p_diff),
    "maps_compared": len(map_diff),
    "max_map_diff": max(map_diff) if map_diff else None,
    "seconds_per_image": [float(np.nanmean(times[:half])), float(np.nanmean(times[half:]))],
}, indent=1))
