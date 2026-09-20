#!/bin/bash
#SBATCH --job-name=vqdm_like
#SBATCH --output=runs/vqdm_like-%j.log
#SBATCH --error=runs/vqdm_like-%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

set -euo pipefail

module load Python/Anaconda_v03.2023

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate vec_quant

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

export PYTHONNOUSERSITE=1
unset PYTHONPATH

if [[ -f "${SLURM_SUBMIT_DIR}/vqdm-sd1-5-optimized.py" ]]; then
  export VQDM_SCRIPT="${SLURM_SUBMIT_DIR}/vqdm-sd1-5-optimized.py"
elif [[ -f "${SCRIPT_DIR}/vqdm-sd1-5-optimized.py" ]]; then
  export VQDM_SCRIPT="${SCRIPT_DIR}/vqdm-sd1-5-optimized.py"
else
  echo "vqdm-sd1-5-optimized.py was not found in ${SLURM_SUBMIT_DIR} or ${SCRIPT_DIR}" >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  exit 1
fi
export VQDM_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

export VQDM_HW_PROFILE="${VQDM_HW_PROFILE:-v100x4}"
export VQDM_SCOPE_SWEEP="${VQDM_SCOPE_SWEEP:-0}"
export VQDM_TARGET_MODE="${VQDM_TARGET_MODE:-all_midblock}"
export VQDM_INCLUDE_CONV="${VQDM_INCLUDE_CONV:-1}"
export VQDM_RUN_ID="${VQDM_RUN_ID:-${VQDM_TARGET_MODE}_${SLURM_JOB_ID:-manual}}"
export VQDM_OUTPUT_DIR="${VQDM_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/runs/vqdm_${VQDM_TARGET_MODE}_${SLURM_JOB_ID:-manual}}"
export VQDM_COMPUTE_HPS=0
export VQDM_FORWARD_METRIC_PROMPTS=2
export VQDM_FORWARD_METRIC_TIMESTEPS=3
export VQDM_FORWARD_METRIC_SEED=20260515

python -V
python -u "${VQDM_SCRIPT}"
