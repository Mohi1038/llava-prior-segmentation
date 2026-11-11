#!/bin/bash
# Rebuild the NeurIPS-format report: stats (CPU, ~15 min) -> figures + numbers.tex -> PDF.
#   bash reports/neurips/build.sh            # figures + PDF (reuses data/stats.json)
#   bash reports/neurips/build.sh --stats    # also recompute data/ from runs/
set -e
cd "$(dirname "$0")/../.."
module load python/3.11 2>/dev/null || true
[ "${1:-}" = "--stats" ] && .venv/bin/python reports/neurips/compute_stats.py
.venv/bin/python reports/neurips/make_figures.py
.venv/bin/python reports/neurips/make_extra_figures.py
cd reports/neurips && /scratch/b23cs1106/tools/tectonic/tectonic -X compile llava_prior_report.tex
cp llava_prior_report.pdf "../../Neurips Styles/llava_prior_report.pdf"
echo "built reports/neurips/llava_prior_report.pdf (copied to Neurips Styles/)"
