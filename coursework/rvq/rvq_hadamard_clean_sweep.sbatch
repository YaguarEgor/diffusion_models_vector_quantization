#!/bin/bash
#SBATCH --job-name=rvq_h_clean
#SBATCH --output=runs/rvq_hadamard_clean-%j.log
#SBATCH --error=runs/rvq_hadamard_clean-%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --constraint="[type_a|type_b|type_c]"

set -euo pipefail

module load Python/Anaconda_v03.2023

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate vec_quant

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

export PYTHONNOUSERSITE=1
unset PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/quantize-vector-hadamard-clean-sweep.py" ]]; then
  export RVQ_SCRIPT="${SCRIPT_DIR}/quantize-vector-hadamard-clean-sweep.py"
elif [[ -f "${SLURM_SUBMIT_DIR}/quantize-vector-hadamard-clean-sweep.py" ]]; then
  export RVQ_SCRIPT="${SLURM_SUBMIT_DIR}/quantize-vector-hadamard-clean-sweep.py"
else
  echo "quantize-vector-hadamard-clean-sweep.py was not found in ${SLURM_SUBMIT_DIR} or ${SCRIPT_DIR}" >&2
  exit 1
fi

if [[ -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_train.csv" && -f "${SLURM_SUBMIT_DIR}/pickscore_filtered_prompts_val.csv" ]]; then
  export PICKSCORE_DATA_DIR="${SLURM_SUBMIT_DIR}"
else
  echo "PickScore CSV files were not found in ${SLURM_SUBMIT_DIR}" >&2
  exit 1
fi
export RVQ_PICKSCORE_DATA_DIR="${PICKSCORE_DATA_DIR}"

# Clean Hadamard sweep: fixed K=256, d=8, M=2/3/4, 10 Linear mid_block layers.
export RVQ_HW_PROFILE="${RVQ_HW_PROFILE:-auto}"
export RVQ_AUTO_MAX_DEVICES=1
export RVQ_FUNNEL_SWEEP=0
export RVQ_STAGE_K_SWEEP=0
export RVQ_K_VALUES=256
export RVQ_HADAMARD_MODE=auto
export RVQ_TARGET_MODE=hardcoded_midblock
export RVQ_INCLUDE_CONV=0
export RVQ_FULL_XTX_RECODE_PASSES=0
export RVQ_COMPUTE_HPS=0
export RVQ_FORWARD_METRIC_PROMPTS=2
export RVQ_FORWARD_METRIC_TIMESTEPS=3
export RVQ_FORWARD_METRIC_SEED=20260515

RVQ_RUN_STEM="rvq_hadamard_clean_${SLURM_JOB_ID:-manual}"
export RVQ_FIXED_SWEEP_OUTPUT_DIR="${RVQ_FIXED_SWEEP_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/runs/${RVQ_RUN_STEM}}"
mkdir -p "${RVQ_FIXED_SWEEP_OUTPUT_DIR}"

echo "RVQ_SCRIPT=${RVQ_SCRIPT}"
echo "RVQ_FIXED_SWEEP_OUTPUT_DIR=${RVQ_FIXED_SWEEP_OUTPUT_DIR}"
echo "RVQ_TARGET_MODE=${RVQ_TARGET_MODE}"
echo "RVQ_INCLUDE_CONV=${RVQ_INCLUDE_CONV}"
echo "RVQ_STAGE_K_SWEEP=${RVQ_STAGE_K_SWEEP}"
echo "RVQ_K_VALUES=${RVQ_K_VALUES}"
echo "RVQ_HADAMARD_MODE=${RVQ_HADAMARD_MODE}"
echo "RVQ_FULL_XTX_RECODE_PASSES=${RVQ_FULL_XTX_RECODE_PASSES}"

python -V
python -u "${RVQ_SCRIPT}"
