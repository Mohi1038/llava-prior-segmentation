"""Self-check for per-layer traces: mean of layers 7..13 must equal the stored aligned map."""

import json
import sys
from pathlib import Path

import numpy as np

from llava_prior.trace import ImageTrace

traces = Path(sys.argv[1])
worst, checked = 0.0, 0
for meta in sorted(traces.glob("*.json"))[:50]:
    if meta.name == "trace_config.json":
        continue
    trace = ImageTrace.load(traces, meta.stem)
    for entry in trace.classes.values():
        if entry.layer_distance is None or entry.distance_aligned is None:
            continue
        worst = max(worst, float(np.abs(entry.layer_distance[7:14].mean(0) - entry.distance_aligned).max()))
        checked += 1
print(json.dumps({"class_traces_checked": checked, "max_abs_diff": worst}))
if checked == 0 or worst > 1e-4:
    raise SystemExit("per-layer self-check FAILED")
print("per-layer self-check passed")
