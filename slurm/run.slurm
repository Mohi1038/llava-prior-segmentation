#!/bin/bash
#SBATCH --job-name=llava-prior
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b23cs1106/projects/llava-prior-segmentation/logs/%x-%j.out
#SBATCH --error=/scratch/b23cs1106/projects/llava-prior-segmentation/logs/%x-%j.out

# Run any shell commands on a GPU node inside the project env.
# Usage: sbatch -J <name> slurm/run.slurm "llava-prior-voc trace --split val --out runs/traces/val"
set -euo pipefail
module purge
module load python/3.11
cd /scratch/b23cs1106/projects/llava-prior-segmentation
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_HOME=/scratch/b23cs1106/hf_cache PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CMD: $1"
eval "$1"
