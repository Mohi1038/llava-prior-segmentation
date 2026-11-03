#!/bin/bash
# Run the authors' released FSeg-LLaVA code (third_party/FSeg-LLaVA, pinned env .venv-official) on a
# VOC image list, then score its VOC21 output (with background) with our evaluator.
# Usage: bash slurm/official.sh <tag> <image-list>
set -euo pipefail
tag=$1; list=$2
out=runs/official/$tag
.venv-official/bin/python third_party/FSeg-LLaVA/infer.py \
  --model-path checkpoints/llava-v1.5-7b \
  --image-root datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages --image-list $list \
  --class-file configs/official_cls_voc20.txt --dataset-format voc --image-aspect-ratio resize \
  --output-dir $out/bg --output-dir-nobg $out/nobg \
  --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth --skip-existing
.venv/bin/python -m llava_prior.voc eval --split-file $list --pred $out/bg
echo OFFICIAL_DONE
