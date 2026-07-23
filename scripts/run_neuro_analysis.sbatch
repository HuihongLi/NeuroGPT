#!/bin/bash
#SBATCH --job-name=spikegpt-neuro
#SBATCH --output=logs/neuro_%j.out          # stdout+stderr (logs/ is created below)
#SBATCH --error=logs/neuro_%j.err
#SBATCH --time=48:00:00                      # walltime; bump if you enlarge the stimulus set
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                    # sklearn/numpy threads; decoder is CPU-bound
#SBATCH --mem=48G
# --- SET THESE for your cluster (misha) -------------------------------------------------
#SBATCH --partition=week                      # TODO: your partition (e.g. day / week / gpu)
# --- GPU (optional) ---------------------------------------------------------------------
# The RNN inference path runs fine on CPU (config run_device: cpu). To use a GPU instead,
# request one here AND set run_device: cuda + float_mode: fp16 in the config.
##SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=huihong.li@yale.edu

set -euo pipefail

# --- environment ------------------------------------------------------------------------
REPO=/gpfs/radev/project/levenstein/hl993/SpikeGPT
PYTHON=/gpfs/radev/home/hl993/miniconda3/bin/python   # the interpreter used interactively
CONFIG=${1:-neuro_analysis/configs/default.yaml}      # optional: pass an alternate config

cd "$REPO"
mkdir -p logs neuro_analysis/outputs

# Keep BLAS/OpenMP threads in line with the allocation (avoids oversubscription).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  config=$CONFIG"
echo "start $(date)"

# --- run --------------------------------------------------------------------------------
# Captures driven + spontaneous conditions, runs the H1-H4 battery on each, then the
# driven-vs-spontaneous contrast. Outputs land under neuro_analysis/outputs/.
srun "$PYTHON" -m neuro_analysis.run_analysis --config "$CONFIG"

echo "done $(date)"
