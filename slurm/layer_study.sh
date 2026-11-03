#!/bin/bash
# Layer study: per-layer trace, then LLaVA-only maps per single layer (no SAM) and layer ranges
# with SAM, then evaluation. Resumable at every step.
# Usage (inside slurm/run.slurm): [MODEL=...] [LAYERS=32] [TRACE_FLAGS=...] bash slurm/layer_study.sh <tag> <split-args...>
#   bash slurm/layer_study.sh dev --split-file splits/voc_train_dev200.txt
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct LAYERS=28 bash slurm/layer_study.sh qwen_dev --split-file splits/voc_train_dev200.txt
set -euo pipefail
tag=$1; shift
model=${MODEL:-llava-hf/llava-1.5-7b-hf}
layers=${LAYERS:-32}
traces=runs/traces/${tag}_layers; pred=runs/pred/${tag}_layers
coarse=configs/voc_layers_coarse.json; ranges=configs/voc_layers_ranges.json
if [ "$layers" != 32 ]; then
  coarse=configs/voc_layers_coarse_${layers}.json; ranges=configs/voc_layers_ranges_${layers}.json
fi
llava-prior-voc trace "$@" --model "$model" --per-layer ${TRACE_FLAGS:-} --out $traces
python scripts/check_layer_traces.py $traces
llava-prior-voc segment "$@" --traces $traces --configs $coarse --out $pred --coarse-only
llava-prior-voc segment "$@" --traces $traces --configs $ranges --out $pred
llava-prior-voc eval "$@" --pred $pred/*/
echo LAYER_STUDY_DONE
